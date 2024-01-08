from nnsight import LanguageModel
import torch

def get_scores(model, hs):
    return model.lm_head(model.transformer.ln_f(hs))

def get_prob_tokens(scores,topk=1):
    probs = torch.nn.functional.softmax(scores, dim=-1)
    favorite_probs, favorite_tokens = probs.topk(k=topk, dim=-1)
    return favorite_probs, favorite_tokens

def load_prefix(path):
    prefix_vector = torch.load(path)
    return prefix_vector.to(torch.float32)

def show_future_lens(model, tok, prefix, context, in_layer = 13, tgt_in_layer = 13, topk = 5, num_toks_gen=4, context_pos = 9, color=None, remote=True):
    from baukit import show

    prefix_pos = len(tok(prefix)['input_ids']) - 1
    
    context = context.detach()
    num_layers = len(model.transformer.h)

    with model.generate(max_new_tokens=num_toks_gen, pad_token_id=50256, remote=remote) as generator:
        with generator.invoke(prefix) as invoker:
            transplant_hs = model.transformer.h[in_layer].output[0].t[prefix_pos].save()
            overall_hs = []
            init_logits = []
            for i in range(num_layers):
                init_logits.append(get_scores(model, model.transformer.h[i].output[0]).save())
                overall_hs.append(model.transformer.h[i].output[0].save())
    
    output = generator.output
    print(tok.decode(output[0][prefix_pos+1:]))

    first_set_logits = [curr_hs.value for curr_hs in init_logits]
    hs = [curr_hs.value for curr_hs in overall_hs]
    first_set_logits = torch.stack(first_set_logits)
    counter = 0
    future_outputs = []
    future_preds = []
    for curr_hs in hs:
        curr_future_outputs = []
        curr_future_preds = []
        for x in range(0, curr_hs.shape[1]):
            sub_hs = curr_hs[:,x,:][None,:]
            with model.generate(max_new_tokens=num_toks_gen, pad_token_id=50256, remote=remote) as generator2:
                with generator2.invoke("_ _ _ _ _ _ _ _ _ _") as invoker:
                    model.transformer.wte.output = context[None,:]
                    model.transformer.h[tgt_in_layer].output[0].t[context_pos] = sub_hs
                    invoker.next()
                    future_output_logits = model.lm_head.output.save()
                    invoker.next()
                    future_output_logits_next = model.lm_head.output.save()
                    invoker.next()
                    future_output_logits_next_next = model.lm_head.output.save()
            counter+= 1
            curr_output = torch.squeeze(future_output_logits.value,0)
            curr_output_next = torch.squeeze(future_output_logits_next.value,0)
            curr_output_next_next = torch.squeeze(future_output_logits_next_next.value,0)

            curr_fav_tok_pred, curr_fav_tok = get_prob_tokens(curr_output[0], topk=1)
            curr_fav_tok_pred_next, curr_fav_tok_next = get_prob_tokens(curr_output_next[0], topk=1)
            curr_fav_tok_pred_next_next, curr_fav_tok_next_next = get_prob_tokens(curr_output_next_next[0], topk=1)
            curr_future_outputs.append([tok.decode(curr_fav_tok),tok.decode(curr_fav_tok_next),tok.decode(curr_fav_tok_next_next)])
            curr_future_preds.append([curr_fav_tok_pred[0].item(), curr_fav_tok_pred_next[0].item(), curr_fav_tok_pred_next_next[0].item()])
        future_outputs.append(curr_future_outputs)
        future_preds.append(curr_future_preds)
        
    # The full decoder head normalizes hidden state and applies softmax at the end.
    favorite_probs, favorite_tokens = get_prob_tokens(first_set_logits, topk=topk)
    
    # All the input tokens.
    prompt_tokens = [tok.decode(t) for t in tok.encode(prefix)]

    # Foreground color shows token probability, and background color shows hs magnitude
    if color is None:
        color = [50, 168, 123]
    def color_fn(p, future_probs = None):
        a = [int(255 * (1-p) + c * p) for c in color]
        if future_probs is not None:
            total_probs = p + sum(future_probs)
            new_p = total_probs / (len(future_probs) + 1)
            a = [int(255 * (1-new_p) + c * new_p) for c in color]
        return show.style(background=f'rgb({a[0]}, {a[1]}, {a[2]})')

    # In the hover popup, show topk probabilities beyond the 0th.
    def hover(tok, prob, toks, fprobs, ftokens):
        total_probs = prob[0] + sum(fprobs)
        new_p = total_probs / (len(fprobs) + 1)
        curr_pred = tok.decode(toks[0]).encode("unicode_escape").decode()
        lines = [f"Average Probability: {new_p}"]
        additional_text = ""
        for curr_fprob, curr_ftoken in zip(fprobs, ftokens):
            curr_ftoken = curr_ftoken.encode("unicode_escape").decode()
            additional_text += f"\n{curr_ftoken}: prob {curr_fprob:.2f}"
        lines.append(f'{curr_pred}: prob {prob[0]:.2f}{additional_text}')
        
        # Commented out code -- add top k token predictions in hover
        # for p, t in zip(prob, toks):
        #     lines.append(f'{tok.decode(t)}: prob {p:.2f}')
        
        return show.attr(title='\n'.join(lines))

    def decode_escape(tok,token,actual_decode=True):
        if not actual_decode:
            if type(token) == list:
                return [t.encode("unicode_escape").decode() for t in token]
            return token.encode("unicode_escape").decode()
        if type(token) == list:
                return [tok.decode(t).encode("unicode_escape").decode() for t in token]
        return tok.decode(token).encode("unicode_escape").decode()
        
    header_line = [ # header line
                [[' ']] + 
             [
                 [show.style(fontWeight='bold', width='50px'), show.attr(title=f'Token {i}'), t]
                 for i, t in enumerate(prompt_tokens)
             ]
         ]
    layer_logits = [
             # first column
             [[show.style(fontWeight='bold', width='50px'), f'L{layer}']] +
             [
                 # subsequent columns
                 [color_fn(p[0], fprobs), hover(tok, p, t, fprobs, ft), show.style(overflowX='hide'), f"{decode_escape(tok, t[0])}{''.join(decode_escape(tok, ft, False))}"]
                 for p, t, ft, fprobs in zip(wordprobs, words, future_words, future_probs)
             ]
        for layer, wordprobs, words, future_words, future_probs in
                zip(range(len(favorite_probs[:, 0])), favorite_probs[:, 0], favorite_tokens[:,0], future_outputs, future_preds)]
    
    show(header_line + layer_logits + header_line)