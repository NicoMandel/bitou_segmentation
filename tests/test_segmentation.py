# to test whether the segmentation configuration of flash works
import pytest


def test_heads():
    from flash.image import ImageClassifier
    backbones = ImageClassifier.available_backbones()
    assert backbones

def test_backbones():
    from flash.image import SemanticSegmentation
    heads = SemanticSegmentation.available_heads()
    assert heads

