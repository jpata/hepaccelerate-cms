CXX      = g++ -fPIC -fopenmp

CXXFLAGS= $(shell root-config --cflags)
LIBS    = $(shell root-config --libs)

SOURCES = WeightCalculatorFromHistogram.cc LeptonEfficiencyCorrector.cc RoccoR.cc
HEADERS = WeightCalculatorFromHistogram.h LeptonEfficiencyCorrector.h RoccoR.h

OBJECTS = $(SOURCES:.cc=.o)

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

libhmm.so: RoccoR.o LeptonEfficiencyCorrector.o WeightCalculatorFromHistogram.o
	$(CXX) $(LIBS) -shared -o $@ RoccoR.o LeptonEfficiencyCorrector.o WeightCalculatorFromHistogram.o
 
# Specifying the object files as intermediates deletes them automatically after the build process.
.INTERMEDIATE:  $(OBJECTS)

.PHONY : clean
clean:
	rm -f libhmm.so
	rm -f $(OBJECTS)
