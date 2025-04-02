from models.models import *

def train_gae(data, gae_model, optimizer, epochs, verbose = False):
    if isinstance(gae_model.encoder, RGCN):
        for e in range(epochs):
            optimizer.zero_grad()
            H_L = gae_model.encode(data.x.float(), data.graph_list)
            loss = gae_model.recon_loss(H_L, data.graph_list[-1].edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
    if isinstance(gae_model.encoder, GCN):
        for e in range(epochs):
            optimizer.zero_grad()
            H_L = gae_model.encode(data.x.float(), data.edge_index)
            loss = gae_model.recon_loss(H_L, data.edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
    print('\n')
    return

def gae_negative_inference(data, model, num_neg):
    inference_dict = dict()
    model.eval()
    if isinstance(model.encoder, RGCN):
        H_L = model.encode(data.x, data.graph_list)

    if isinstance(model.encoder, GCN):
        H_L = model.encode(data.x.float(), data.edge_index)

    for element in data.U:
        dist = torch.cdist(H_L[element].unsqueeze(0), H_L[data.P])
        value = dist.mean()
        inference_dict[element] = value
    dicionario_ordenado = dict(sorted(inference_dict.items(),reverse=True, key=lambda item: item[1]))
    return torch.stack(list(dicionario_ordenado.keys())[:num_neg])
