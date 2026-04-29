import unittest


class PEFTAdapterTests(unittest.TestCase):
    def test_peft_adapter_reports_missing_dependency(self):
        from castfactory.models.adapters import PEFTAdapter

        adapter = PEFTAdapter(method="lora", r=8)

        try:
            import peft  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(ImportError, "peft"):
                adapter.apply(model=object())
        else:
            self.assertEqual(adapter.method, "lora")


if __name__ == "__main__":
    unittest.main()
