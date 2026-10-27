"""Focused tests for data transformations and partition leakage guards."""

from pathlib import Path
import tempfile
import unittest

from freeze_dataset import assert_disjoint, assign_partitions, transform_row
from materialize_dataset import read_checksums


def source_line(box=(.5, .5, .4, .4), points=None, cid=4):
    points = points or [(i/10, .5) for i in range(9)]
    return ' '.join(str(v) for v in [cid, *box, *(x for p in points for x in p)])


class PolicyTests(unittest.TestCase):
    def test_clipped_box_shared_by_tasks(self):
        det, pose, info = transform_row(source_line(box=(0., .5, .4, .4)))
        self.assertEqual(det.split(), pose.split()[:5])
        self.assertAlmostEqual(float(det.split()[1]), .1)
        self.assertAlmostEqual(float(det.split()[3]), .2)
        self.assertTrue(info['box_clipped'])

    def test_mask_not_clamp(self):
        pts = [(-.1, .5), (0., 0.), (1., 1.)] + [(.2, .3)]*6
        _, pose, info = transform_row(source_line(points=pts))
        v = pose.split()
        self.assertEqual(v[5:8], ['0.000000000', '0.000000000', '0'])
        self.assertEqual(v[8:11], ['0.000000000', '0.000000000', '1'])
        self.assertEqual(v[11:14], ['1.000000000', '1.000000000', '1'])
        self.assertEqual(info['masked_points'], 1)

    def test_class_and_valid_point_order_preserved(self):
        pts = [(.8-i*.05, .2+i*.02) for i in range(9)]
        det, pose, _ = transform_row(source_line(points=pts))
        self.assertEqual(det.split()[0], '4')
        values = list(map(float, pose.split()[5:]))
        for i, (x, y) in enumerate(pts):
            self.assertAlmostEqual(values[i*3], x)
            self.assertAlmostEqual(values[i*3+1], y)
        self.assertEqual(len(pose.split()), 32)

    def test_invalid_source_is_not_silently_dropped(self):
        for text in ['0 1 2', source_line(box=(2, .5, .1, .1)), source_line(box=(.5, .5, 0, .1)), source_line(cid=5), source_line().replace('0.5', 'nan', 1)]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    transform_row(text)

    def rows(self):
        return [{'origin': 'trainval', 'stem': f'train_{i}_0_0', 'scene': f'scene_{i}',
                 'source_image_sha256': f'hash_{i}', 'classes': {str(k): 2 for k in range(5)}} for i in range(10)] + [
                 {'origin': 'test', 'stem': 'test_0_0', 'scene': 'test_scene',
                  'source_image_sha256': 'test_hash', 'classes': {str(k): 1 for k in range(5)}}]

    def test_partition_determinism_and_source_scene_blocks(self):
        a, b = self.rows(), self.rows()
        assign_partitions(a, set(), 42, .2, 100)
        assign_partitions(b, set(), 42, .2, 100)
        self.assertEqual(a, b)
        assert_disjoint(a)
        self.assertEqual(sum(r['split'] == 'val' for r in a), 2)

    def test_exact_duplicate_scenes_cannot_straddle_train_val(self):
        rows = self.rows()
        rows[1]['source_image_sha256'] = rows[0]['source_image_sha256']
        assign_partitions(rows, set(), 42, .2, 100)
        self.assertEqual(rows[0]['split'], rows[1]['split'])
        assert_disjoint(rows)

    def test_duplicate_of_test_quarantines_training_scene(self):
        rows = self.rows()
        rows[0]['source_image_sha256'] = rows[-1]['source_image_sha256']
        assign_partitions(rows, set(), 42, .2, 100)
        self.assertEqual(rows[0]['split'], 'quarantine')
        self.assertEqual(rows[-1]['split'], 'test')
        assert_disjoint(rows)

    def test_uncertain_class_quarantine_keeps_test_label_unchanged(self):
        rows = self.rows()
        assign_partitions(rows, {'test/test_0_0.txt'}, 42, .2, 100)
        self.assertEqual(rows[-1]['split'], 'quarantine')
        self.assertEqual(rows[-1]['classes']['4'], 1)

    def test_overlap_guard(self):
        rows = self.rows()
        assign_partitions(rows, set(), 42, .2, 100)
        rows[-1]['source_image_sha256'] = next(r['source_image_sha256'] for r in rows if r['split'] == 'train')
        with self.assertRaises(ValueError):
            assert_disjoint(rows)

    def test_frozen_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.csv').write_text('tampered')
            (root/'SHA256SUMS').write_text('0'*64+'  manifest.csv\n')
            with self.assertRaises(ValueError):
                read_checksums(root)


if __name__ == '__main__':
    unittest.main()
