import unittest
from unittest.mock import patch
import torch
from torch import nn
from examples.synthetic_episode import make_episode, make_model
from topo_opd import ToPoOPD, masked_average_pool
from topo_opd.backbones.dgcnn import _edge_features


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(0)
        self.model = make_model().eval()

    def test_all_protocols_train_eval_backward(self):
        for way in (2, 3):
            for shot in (1, 5):
                with self.subTest(way=way, shot=shot):
                    args = make_episode(way, shot)
                    self.model.train()
                    out = self.model(*args)
                    self.assertEqual(out['logits'].shape, (2, 15, way + 1))
                    self.assertEqual(out['shot_initial_prototypes'].shape, (2, way * shot, way + 1, 8))
                    self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
                    self.model.zero_grad()
                    (out['logits'].square().mean() + out['opd_loss']).backward()
                    grads = [p.grad for p in self.model.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                    self.model.eval()
                    self.assertNotIn('opd_loss', self.model(*args))

    def test_legacy_equals_explicit_single_cloud(self):
        p, m, q = make_episode(2, 1)
        # Legacy permits absent classes; explicit input needs all classes represented.
        m = m[:, :1, :2]
        a = self.model(p[:, 0], m[:, 0], q)
        b = self.model(p[:, :1], m, q)
        for key, value in a.items():
            torch.testing.assert_close(value, b[key])

    def test_legacy_matches_original_computation_including_empty_class(self):
        p, m, q = make_episode(3, 1)
        p, m = p[:, 0], m[:, 0]
        f = self.model.backbone(p)
        initial = masked_average_pool(f, m)
        topo = self.model.topo(initial, f, p[..., :3], m)
        expected = self.model.logit_scale * torch.einsum(
            'bnd,bcd->bnc', torch.nn.functional.normalize(self.model.backbone(q), dim=-1),
            torch.nn.functional.normalize(topo.calibrated, dim=-1))
        out = self.model(p, m, q)
        torch.testing.assert_close(out['logits'], expected)
        torch.testing.assert_close(out['calibrated_prototypes'], topo.calibrated)

    def test_equal_shot_mean_not_point_mass_mean(self):
        model = ToPoOPD(nn.Identity(), 3, topology_dim=4, topology_neighbors=2).eval()
        p = torch.tensor([[[[2., 0, 0], [99., 0, 0]], [[8., 0, 0], [8., 0, 0]]]])
        m = torch.tensor([[[[1., 0]], [[1., 1]]]])
        out = model(p, m, p[:, 0])
        torch.testing.assert_close(out['initial_prototypes'], torch.tensor([[[5., 0, 0]]]))
        torch.testing.assert_close(out['calibrated_prototypes'], out['shot_calibrated_prototypes'].mean(1))
        m[:, 0] = 0
        out = model(p, m, p[:, 0])
        torch.testing.assert_close(out['initial_prototypes'], torch.tensor([[[8., 0, 0]]]))
        torch.testing.assert_close(out['calibrated_prototypes'], out['shot_calibrated_prototypes'][:, 1])

    def test_shot_permutation_and_cloud_isolation(self):
        p, m, q = make_episode(3, 5)
        out = self.model(p, m, q)
        perm = torch.randperm(p.shape[1])
        other = self.model(p[:, perm], m[:, perm], q)
        torch.testing.assert_close(out['logits'], other['logits'])
        # Compare actual DGCNN and ToPo results against separate cloud calls.
        flat = p.flatten(0, 1)
        features = self.model.backbone(flat)
        for i in range(flat.shape[0]):
            torch.testing.assert_close(features[i:i+1], self.model.backbone(flat[i:i+1]))
        for s in range(p.shape[1]):
            single = self.model(p[:, s], m[:, s], q)
            valid = out['shot_valid'][:, s]
            torch.testing.assert_close(out['shot_calibrated_prototypes'][:, s][valid], single['calibrated_prototypes'][valid])
        changed = p.clone()
        changed[:, -1] += 100
        altered = self.model(changed, m, q)
        torch.testing.assert_close(out['shot_calibrated_prototypes'][:, :-1], altered['shot_calibrated_prototypes'][:, :-1])

    def test_knn_does_not_cross_batch_clouds(self):
        p = torch.tensor([[[0.], [10.]], [[0.1], [10.1]]])
        edges = _edge_features(p, 2)
        self.assertEqual(set(edges[..., 0].flatten().tolist()), {-10., 0., 10.})
        # Point-axis concatenation introduces cross-cloud edges for these clouds.
        mixed = _edge_features(p.reshape(1, 4, 1), 2)
        self.assertTrue((mixed[..., 0].abs() > 0).any())
        self.assertTrue((mixed[..., 0].abs() < 1).all())

    def test_class_masks_pooling(self):
        f = torch.tensor([[[1., 2.], [3., 4.], [100., 200.]]])
        m = torch.tensor([[[1., 1., 0.], [0., 0., 1.], [0., 0., 0.]]])
        torch.testing.assert_close(masked_average_pool(f, m), torch.tensor([[[2., 3.], [100., 200.], [0., 0.]]]))

    def test_invalid_episode_rejected_before_encoding(self):
        p, m, q = make_episode(2, 1)
        invalid = [(p, m[..., :-1], q), (p, m, q[:1]), (p, m * 0, q),
                   (p, m - 1, q), (p, m * float('nan'), q), (p[:, :0], m[:, :0], q)]
        for args in invalid:
            with self.subTest(shape=args[0].shape), patch.object(self.model.backbone, 'forward', side_effect=AssertionError('encoded invalid input')):
                with self.assertRaises(ValueError):
                    self.model(*args)


if __name__ == '__main__':
    unittest.main()
