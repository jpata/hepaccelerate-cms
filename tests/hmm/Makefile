all: libhmm.so

# We need to be careful to link against the intel version of OpenMP to prevent segfaults
# https://stackoverflow.com/questions/25986091/telling-gcc-to-not-link-libgomp-so-it-links-libiomp5-instead
CXX      = g++ -fPIC -O3
LINK     = g++ -fPIC -L/usr/local/anaconda3/lib -liomp5 -lmkl_intel_thread

CXXFLAGS= $(shell root-config --cflags)
LIBS    = $(shell root-config --libs)

SOURCES = GBRForestTools.cc LeptonEfficiencyCorrector.cc RoccoR.cc tinyxml2.cc WeightCalculatorFromHistogram.cc MiscVariables.cc NNLOPSReweighting.cc hRelResolution.cc ZpTReweighting.cc
HEADERS = *.h

OBJECTS = $(SOURCES:.cc=.o)

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

libhmm.so: $(OBJECTS)
	$(LINK) $(LIBS) -shared -o $@ $^

# Specifying the object files as intermediates deletes them automatically after the build process.
.INTERMEDIATE:  $(OBJECTS) tmva

.PHONY : clean
clean:
	rm -f libhmm.so
	rm -f tmva
	rm -f $(OBJECTS)
