from models.redundant import RedTwoLayerFC, RedFourLayerFC, RedFCLeNet5, RedFCVGG #, RedFCResNet18
from models.wide import WideTwoLayerFC, WideFourLayerFC, WideFCLeNet5 #, WideFCVGG, WideFCResNet18
from models.baseline import TwoLayerFC, FourLayerFC, LeNet5, VGG #, ResNet18

__all__ = [
    'RedTwoLayerFC',
    'RedFourLayerFC',
    'RedFCLeNet5',
    'RedFCVGG',
#    'RedFCResNet18',
    'WideTwoLayerFC',
    'WideFourLayerFC',
    'WideFCLeNet5',
#    'WideFCVGG',
#    'WideFCResNet18',
    'TwoLayerFC',
    'FourLayerFC',
    'LeNet5',
    'VGG'
#    'ResNet18'
]


