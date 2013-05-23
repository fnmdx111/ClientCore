Client Core for Secure Image Retrieval System
=============================================

Requirement
-----------

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

* adjust algorithm so that difference between original and retrieved is not too great


Note
----

After changing numpy.float32 to numpy.float64, key space has increased from 10 ** 6 to 10 ** 16.
If we change the function f from 1 - 2 * x * x to n * x * (1 - x), the key space can be increased by another 10 ** 16.
However, the chance that two keys generate the same sequence is still unknown.

And this encryption method (permutation) is prone to known-plaintext attack.

* added 13/05/08 Maybe images need not be idct-ed to be able to see (snow and mosaic), just send the encrypted file to server, and in this way block cipher could be used.

License
-------

THIS PROJECT IS LICENSED UNDER GPL.


contact: chsc4698@gmail.com

