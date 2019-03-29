Accelerated array analysis on CMS NanoAOD data. Requires Python 3, Numba, uproot, awkward-array, optionally CUDA.
Use the provided singularity image to get started fast.

~~~
git clone git@github.com:jpata/hepaccelerate.git
cd hepaccelerate

#prepare a list of files (currently must be on the local filesystem, not on xrootd) to read
#replace /nvmedata with your local location of ROOT files
find /nvmedata/store/mc/RunIIFall17NanoAODv4/GluGluHToMuMu_M125_*/NANOAODSIM -name "*.root | head -n100 > filelist.txt

#Download the uproot+cupy+numba singularity image from CERN
wget https://jpata.web.cern.ch/jpata/singularity/cupy.simg -o singularity/cupy.simg

##or on lxplus
#cp /eos/user/j/jpata/www/singularity/cupy.simg ./singularity/
##or compile yourself if you have ROOT access on your machine
#cd singularity;make

#Run the test script using singularity (currently singularity does NOT work on lxplus)
#In case you get errors with data not being found, use the option -B to bind external mounted disks to the singularity image (in my case /nvmedata) in case
LC_ALL=C PYTHONPATH=./ singularity exec -B /nvmedata --nv singularity/cupy.simg python3 tests/simple.py --filelist filelist.txt

#output will be stored in this json
cat out.json

#second time around, can load the data from the cache, which is much faster
LC_ALL=C PYTHONPATH=./ singularity exec -B /nvmedata --nv singularity/cupy.simg python3 tests/simple.py --filelist filelist.txt --from-cache

#use CUDA for array processing
LC_ALL=C PYTHONPATH=./ singularity exec -B /nvmedata --nv singularity/cupy.simg python3 tests/simple.py --filelist filelist.txt --from-cache --use-cuda
~~~
