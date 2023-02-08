Currently supported benchmarks or `circuit_type`s are
    - qft
    - tfim

Currently supported `connectivity` types are
    - linear
    - complete
Note that as only blocksize 3 is used, this is exhaustive.

Start by generating the benchmarks with 
    $ python {circuit_type}/generate.py
These benchmarks must then be partitioned.


Take the file called `{circuit_type}_list_{smallest_size}_{largest_size}.pickle`
and place it into the `instantiations` directory. Go into this directory. Next, 
partition each circuit by calling 
    $ python partition.py {circuit_type}_list_{smallest_size}_{largest_size}.pickle

Find templates that instantiate each partition in each circuit by calling 
    $ python instantiate.py {circuit_type} {connectivity}

Find try to find the outlier templates by moving into the `statistics` directory.
call
    $ python stats.py {circuit_type} {connectivity}
to generate a file called `reinstantiate-{circuit_type}_{connectivity}.pickle`.
This contains block numbers that should be retried. The threshold for what is 
considered an outlier can be changed by altering the `outlier_threshold` var-
iable in the `stats.py` file. Move this file into the `reinstantiations` direc-
tory. Call
    $ python reinstantiate.py {circuit_type} {connectivity}

Move into the `final_template_assignments` directory and call
    $ python combine.py {circuit_type} {connectivity}