import types
import torch


def _fake_hfds_module(monkeypatch):
    class FakeSplit:
        def __init__(self, n=4):
            self._n = n
        def __len__(self):
            return self._n
        def __getitem__(self, idx):
            # 3x8x8 tensor image and 5 captions
            img = torch.zeros(3, 8, 8)
            return {"image": img, "sentences_raw": [f"cap{i}-{idx}" for i in range(5)]}

    class FakeDataset:
        def __init__(self):
            self.train = FakeSplit()
            self.validation = FakeSplit()
        def __getitem__(self, key):
            return getattr(self, key)

    def fake_load_dataset(name, cache_dir=None):
        return FakeDataset()

    import src.data.dataloader as dl
    # Patch inside module import path
    dl.hfds = types.SimpleNamespace(load_dataset=fake_load_dataset)
    return dl


def test_make_coco_dataloader(monkeypatch):
    dl = _fake_hfds_module(monkeypatch)
    loader = dl.make_coco_dataloader(split="train", batch_size=2, num_workers=0)
    batch = next(iter(loader))
    images, captions = batch
    assert isinstance(images, list) and isinstance(captions, list)
    assert len(images) == 2 and len(captions) == 2


def test_coco_collate_lists(monkeypatch):
    dl = _fake_hfds_module(monkeypatch)
    # Build a batch manually
    img = torch.zeros(3, 8, 8)
    batch = [(img, "a"), (img, "b")]
    images, captions = dl.coco_collate(batch)
    assert isinstance(images, list) and isinstance(captions, list)
    assert len(images) == 2 and captions == ["a", "b"]
