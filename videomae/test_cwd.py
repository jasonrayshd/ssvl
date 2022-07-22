from epickitchens_utils import extract_zip

path_to_save="/data/shared/ssvl/epic-kitchens50/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P01/P01_13"

extract_zip(path_to_save, ext="tar", frame_list = ["frame_{:010d}.jpg".format(i) for i in range(1, 5)], flow=True)
