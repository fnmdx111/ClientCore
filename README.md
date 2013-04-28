Client Core for Secure Image Retrieval System
=============================================

Required Library
----------------

* opencv - image manipulation
* numpy - backend for opencv
* requests - communication between client and server


Goal
----

* permutation transformation on given image
* send transformed image to server
* parse retrieved information from server


TODO
----

* extend key space by introducing three float key,
one for row permutation,
one for column permutation,
one for block permutation


Note
----

After changing numpy.float32 to numpy.float64, key space has increased from 10 ** 6 to 10 ** 16.
If we change the function f from 1 - 2 * x * x to n * x * (1 - x), the key space can be increased by another 10 ** 16.
However, the chance that two keys generate the same sequence is still unknown.


License
-------

THIS PROJECT IS LICENSED UNDER GPL.


contact: chsc4698@gmail.com

