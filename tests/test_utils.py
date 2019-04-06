from hepaccelerate.utils import JaggedStruct
import numpy as np

def test_jaggedstruct():
    js = JaggedStruct([0,2,3], {"pt": [0,1,2,3,4,5]}, np)
    attr_names_dtypes = [("offsets", "uint64"), ("pt", "float32")]
    js.attr_names_dtypes = attr_names_dtypes
    js.save("cache")

    js2 = JaggedStruct.load("cache", attr_names_dtypes, np)

    np.all(js.offsets == js2.offsets)
    for k in js.attrs_data.keys():
        np.all(getattr(js, k) == getattr(js2, k))

test_jaggedstruct()