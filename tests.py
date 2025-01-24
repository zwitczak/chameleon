import unittest
from clustering import Chameleon
import numpy as np

class ChameleonTest(unittest.TestCase):
    def test_class_initialization(self):
        # Create a dummy dataset
        dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        # Initialize Chameleon with the dummy dataset
        chameleon = Chameleon(dataset=dataset, min_size=0.1, k_neighbors=3)

        # Check if the attributes are correctly initialized
        self.assertTrue(np.array_equal(chameleon.dataset, dataset), "Dataset initialization failed.")
        self.assertEqual(chameleon.k_neighbors, 3, "k_neighbors initialization failed.")
        self.assertEqual(chameleon.min_cluster_size, int(0.1 * len(dataset)), "min_cluster_size initialization failed.")
        self.assertIsNone(chameleon.graph, "Graph should be initialized as None.")
        self.assertIsNone(chameleon.clusters, "Clusters should be initialized as None.")
        self.assertIsNone(chameleon.scheme, "Scheme should be initialized as None.")
        self.assertIsNone(chameleon.alpha, "Alpha should be initialized as None.")
        self.assertIsNone(chameleon.min_rc, "min_rc should be initialized as None.")
        self.assertIsNone(chameleon.mic_ri, "mic_ri should be initialized as None.")


if __name__ == '__main__':
    unittest.main()