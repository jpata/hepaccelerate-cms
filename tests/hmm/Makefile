all: libhmm.so

CXX      = g++ -fPIC -fopenmp -g

CXXFLAGS= $(shell root-config --cflags)
LIBS    = $(shell root-config --libs)

SOURCES = GBRForestTools.cc LeptonEfficiencyCorrector.cc RoccoR.cc tinyxml2.cc WeightCalculatorFromHistogram.cc
HEADERS = *.h

OBJECTS = $(SOURCES:.cc=.o)

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

libhmm.so: $(OBJECTS)
	$(CXX) $(LIBS) -shared -o $@ $^

# Specifying the object files as intermediates deletes them automatically after the build process.
.INTERMEDIATE:  $(OBJECTS)

.PHONY : clean
clean:
	rm -f libhmm.so
	rm -f $(OBJECTS)
