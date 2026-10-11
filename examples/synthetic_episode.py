"""CPU smoke run: python -m examples.synthetic_episode."""
import torch
from topo_opd import DGCNNBackbone, ToPoOPD


def make_episode(way, shot, batch=2, points=12):
    """One distinct cloud per foreground class/shot; channel 0 is background.

    Real loaders must enforce scene/block IDs and train/evaluation splits.
    """
    generator = torch.Generator().manual_seed(way * 100 + shot)
    support = torch.randn(batch, way * shot, points, 6, generator=generator)
    masks = torch.zeros(batch, way * shot, way + 1, points)
    for c in range(way):
        for k in range(shot):
            masks[:, c * shot + k, 0, :points // 2] = 1
            masks[:, c * shot + k, c + 1, points // 2:] = 1
    query = torch.randn(batch, points + 3, 6, generator=generator)
    return support, masks, query


def make_model():
    return ToPoOPD(DGCNNBackbone(edge_dims=(8, 8), output_dim=8, k=3),
                   feature_dim=8, topology_dim=8, topology_neighbors=3,
                   max_topology_points=8, opd_embedding_dim=8)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for way in (2, 3):
        for shot in (1, 5):
            model = make_model().train()
            output = model(*make_episode(way, shot))
            (output['logits'].square().mean() + output['opd_loss']).backward()
            assert all(torch.isfinite(v).all() for v in output.values())
            print(f'{way}-way {shot}-shot: logits={tuple(output["logits"].shape)}, backward OK')


if __name__ == '__main__':
    main()
