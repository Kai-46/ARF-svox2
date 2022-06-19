# ARF: Artistic Radiance Fields

Project page: <https://www.cs.cornell.edu/projects/arf/>

![](./resources/ARF.mov)


Citation:
```
@misc{zhang2022arf,
      title={ARF: Artistic Radiance Fields}, 
      author={Kai Zhang and Nick Kolkin and Sai Bi and Fujun Luan and Zexiang Xu and Eli Shechtman and Noah Snavely},
      year={2022},
      booktitle={arXiv},
}
```

## Quick start

### Install environment
```bash
. ./create_env.sh
```
### Download data
```bash
. ./download_data.sh
```
### Optimize artistic radiance fields
```bash
cd opt && . ./try_{llff/tnt/custom}.sh [scene_name] [style_id]
```
* Select ```{llff/tnt/custom}``` according to your data type. For example, use ```llff``` for ```flower``` scene, ```tnt``` for ```Playground``` scene, and ```custom``` for ```lego``` scene. 
* ```[style_id].jpg``` is the style image inside ```./data/styles```. For example, ```14.jpg``` is the starry night painting.
* Note that a photorealistic radiance field will first be reconstructed for each scene, if it doesn't exist on disk. This will take extra time.

### Check results
The optimized artistic radiance filed is inside ```opt/ckpt_arf/[scene_name]_[style_id]```, while the photorealistic one is inside ```opt/ckpt_svox2/[scene_name]```.

### Custom data
Please follow the steps on [Plenoxel](https://github.com/sxyu/svox2)  to prepare your own custom data.

## ARF with other NeRF variants
* [ARF-TensoRF](): to be released; stay tuned.
* [ARF-NeRF](): to be released; stay tuned.

## Acknowledgement:
We would like to thank [Plenoxel](https://github.com/sxyu/svox2) authors for open-sourcing their implementations.
