
* See `usage_example`.

* [Related reference](https://urldefense.com/v3/__https://iacl.ece.jhu.edu/index.php/Retinal_layer_segmentation_of_macular_OCT_images__;!!Mi0JBg!O4aT4ZWGclDtJ8KVVE39vajV7hask9zr4ZnN-YaQYQJUBetLVLFiQPV3gdRqm1946KxpvlQARLrqr58$).
  * [paper](https://doi.org/10.1364/BOE.4.001133)

* `test_optic_disc_detection`:
  * TODO: requires evaluation on a set of at least 100 OCT images to see how 
  exactly this simple method can determine the center of optic disc.

---
**Original version was found at [here](https://github.com/zhiqiiiiiii/OCT_preprocess)**  with the following readme:


This repository is adapted from [Weiting Tan](https://github.com/steventan0110/OCT_preprocess):

* `read_img.py`: read Cirrus .img file to numpy array.

* `detect_retina`: detect boundaries for ILM, IS/OS, and BM. Then the volumn is flatterned to BM.

* `inpaint_nan3.py`: fill in values for nan points. 

Usage:

```python
from read_img import read_img
from detect_retina import detect_retina 
img_vol = read_img(filepath, pixelsX, pixelsY, pixelsZ)
vol_flatterned, retina_mask, upper_bound, lower_bound = detect_retina(img_vol)
```
