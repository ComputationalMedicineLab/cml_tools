"""Import all modules - basic sanity check against broken imports"""
import unittest

class TestImports(unittest.TestCase):
    def test_imports(self):
        try:
            import cml_tools
            import cml_tools.fastica
            import cml_tools.online_norm
            import cml_tools.testing
            import cml_tools.whiten

            import cml_tools.plot
            import cml_tools.plot.ehr_counts
            import cml_tools.plot.err_interval
            import cml_tools.plot.loss_curves

            import cml_tools.nn.dataset
            import cml_tools.nn.interpolate
            import cml_tools.nn.loss_functions
            import cml_tools.nn.modules
            import cml_tools.nn.online_norm
            import cml_tools.nn.trainers
        except ImportError as exc:
            raise self.failureException from exc

if __name__ == '__main__':
    unittest.main()
