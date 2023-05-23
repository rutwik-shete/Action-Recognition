def getTop5Accu(logits,label):

    logits = logits.argsort(dim=1, descending=True)
    preds = logits[:,:5]

    correct = 0
    
    for idx, pred in enumerate(preds):
        if label[idx] in pred:
            correct += 1

    return correct