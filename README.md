![logo](doc/logo.png)

# pyImageAugmentation

Image augumentation for deep learning.



## Examples of usage

Renames all image files to SHA-1 format (sha1.jpeg/png)

`./pyImageAug.py -i test/`



Renames all image files to SHA-1 format (sha1.jpeg/png) and creates new augmented files based on original files with random transformations like

- gauss noise added
- rotatation
- translation
- perspective transformation
- blur
- flip horizontally
- affine transformation
- hue
- saturation
- contrast
- brightness
- mirror and mosaic
- night
- vignette
- rain

and mixes of all these methods.

`./pyImageAug.py -i test/ -a`

## Help

```shell
usage: pyImageAug.py [-h] [-i INPUT] [-ar] [-v]

optional arguments:
  -h, --help       show this help message and exit
  -i INPUT, --input INPUT Input path
  -ar, --augmentation  Process extra image augmentation.
  -v, --verbose     Show verbose finded and processed data
```
