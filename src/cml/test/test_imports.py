"""Import all modules - basic sanity check against broken imports"""
import unittest

class TestImports(unittest.TestCase):
    def test_imports(self):
        try:
            import cml
            import cml.databricks
            import cml.fastica
            import cml.online_norm
            import cml.whiten

            import cml.ehr.code_name_mangle

            import cml.nn.dataset
            import cml.nn.interpolate
            import cml.nn.loss_functions
            import cml.nn.modules
            import cml.nn.online_norm
            import cml.nn.trainers

            import cml.plot
            import cml.plot.ehr_counts
            import cml.plot.err_interval
            import cml.plot.ica_model
            import cml.plot.loss_curves

            import cml.script.fastica
        except ImportError as exc:
            raise self.failureException from exc

if __name__ == '__main__':
    unittest.main()
