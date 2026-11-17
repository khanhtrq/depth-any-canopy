"""Exercise callback persistence without requiring the GPU training stack."""
import ast
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


source = Path(__file__).resolve().parents[1] / 'training_curves.py'
node = next(n for n in ast.parse(source.read_text(encoding='utf-8-sig')).body if isinstance(n, ast.ClassDef))
namespace = {'Callback': object, 'Path': Path, 'csv': csv, 'Figure': Mock(), 'FigureCanvasAgg': Mock()}
axes = [Mock() for _ in range(4)]
namespace['Figure'].return_value.subplots.return_value.ravel.return_value = axes
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
TrainingCurves = namespace['TrainingCurves']


class CurveTests(unittest.TestCase):
    def test_epoch_history_and_rank_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = TrainingCurves(directory)
            scalar = Mock()
            scalar.detach.return_value.clone.return_value = scalar
            scalar.cpu.return_value.item.return_value = 0.25
            trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False,
                current_epoch=0, global_step=10,
                callback_metrics={'train_Loss(MSE)_epoch': scalar, 'val_Loss(MSE)_epoch': scalar},
                strategy=SimpleNamespace(reduce=Mock(return_value=scalar)),
                optimizers=[SimpleNamespace(param_groups=[{'lr': 1e-6}])])
            callback.on_fit_start(trainer, None)
            callback.on_train_epoch_end(trainer, None)
            trainer.current_epoch = 1
            callback.on_train_epoch_end(trainer, None)
            with (Path(directory) / 'metrics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['epoch'] for row in rows], ['1', '2'])
            self.assertEqual(rows[0]['val_Loss(MSE)'], '0.25')
            namespace['Figure'].return_value.savefig.assert_called_with(Path(directory) / 'training_curves.png', dpi=160)
            trainer.is_global_zero = False
            before = trainer.strategy.reduce.call_count
            callback.on_train_epoch_end(trainer, None)
            self.assertEqual(len(callback.rows), 2)
            self.assertEqual(trainer.strategy.reduce.call_count, before + 2)
            trainer.sanity_checking = True
            callback.on_train_epoch_end(trainer, None)
            self.assertEqual(trainer.strategy.reduce.call_count, before + 2)


if __name__ == '__main__':
    unittest.main()
