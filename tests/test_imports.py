"""Import all modules - basic sanity check against broken imports"""
import unittest

class TestImports(unittest.TestCase):
    def test_imports(self):
        try:
            import cml_tools
            import cml_tools.online_norm

            import cml_tools.plot
            import cml_tools.plot.ehr_counts
            import cml_tools.plot.err_interval
            import cml_tools.plot.loss_curves

            import cml_tools.neural_net.dataset
            import cml_tools.neural_net.interpolate
            import cml_tools.neural_net.loss_functions
            import cml_tools.neural_net.modules
            import cml_tools.neural_net.online_norm
            import cml_tools.neural_net.testing
            import cml_tools.neural_net.trainers
            import cml_tools.neural_net.whiten
        except ImportError as exc:
            raise self.failureException from exc

if __name__ == '__main__':
    unittest.main()
