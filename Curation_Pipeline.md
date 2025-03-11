# Curation Pipeline

Our WebVi3D curation pipeline contains 4 steps, including:

- Temporal-Spatial Downsampling
- Semantic-Based Dynamic Recognition 
- Non-Rigid Dynamic Filtering 
- Tracking-Based Small Viewpoint Filtering

## Running

We provide some unfiltered video cases and the demo code of data curation pipeline for your own video dataset. Notice that you may modify the code into parallel processing for faster processing on larger scale dataset. You may also adjust the parameters in each step for your own applications.

First, install extra dependencies:
```sh
cd ./WebVi3D
pip install -r requirements.txt
```

### Down Sampling and Dynamic Recognition

```sh
python s1_downsample.py --input_dir <video folder> --output_dir <output dir for seperated frames> --txt <output txt file for indexing>
```

You will get all video names in the `txt` file, and the processed frames are save in `output_dir`, the filtered videos will not be saved in this step.

### Flow Generation

We use RAFT to generate optical flow for dynamic masks generation.

```sh
python s2_generate_flow_earlycheck.py --dataset_path <seperated frames from last step> --txt <output txt file for further indexing>
```

The computed forward and backward flow will be saved in `flow` subdirectory for further processing. Notice that we do early checking that filter out the videos with very small optical flow values in the earlier frames.

### Dynamic Masking 

We generate dynamic masks to filter out videos with moving things, run:

````sh
python s3_generate_mask.py --dataset_path <seperated frames and flow from last step> --txt <output txt file for further indexing> 
````

to obtain dynamic motion mask, where the video name will be stored in txt.

### Dynamic and Small Camera Movement Filtering

We then filter our dynamic scenes and small camera movements in one time, and save the final results to txt. Similarly, run:

```sh
python s4_filter.py --dataset_path <seperated frames, flow and mask from the last step>  --mask_done_file <dynamic mask txt file from the last step> --txt <final output video names>
```

The filtered video names will be listed in the final txt file.


### Acknowledgements
We appreciate [RAFT](https://github.com/princeton-vl/RAFT), [Co-Tracker](https://github.com/facebookresearch/co-tracker) for their awesome open-sourced projects.
