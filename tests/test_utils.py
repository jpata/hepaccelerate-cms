from hepaccelerate.utils import JaggedStruct
import numpy as np

def test_jaggedstruct():
    attr_names_dtypes = [("Muon_pt", "float64")]
    js = JaggedStruct([0,2,3], {"pt": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])}, "Muon_", np, attr_names_dtypes)
    js.attr_names_dtypes = attr_names_dtypes
    js.save("cache")

    js2 = JaggedStruct.load("cache", "Muon_", attr_names_dtypes, np)

    np.all(js.offsets == js2.offsets)
    for k in js.attrs_data.keys():
        np.all(getattr(js, k) == getattr(js2, k))

if __name__ == "__main__":
    test_jaggedstruct()
