#!/usr/bin/env python

def prepend_files(prepend, target_dir, glob_search='*.png'):
    import glob,os,shutil
    target_dir = os.path.expanduser(target_dir)
    file_paths = glob.glob(os.path.join(target_dir, glob_search))
    fnames = [os.path.split(f)[1] for f in file_paths]
    new_fnames = [os.path.join(target_dir, prepend + old) for old in fnames]
    for old,new in zip(file_paths, new_fnames):
        print 'Moving: {} to {}'.format(os.path.split(old)[1], os.path.split(new)[1])
        shutil.move(old, new)

def frames_to_video(frame_dir, video_fname = None, video_fps = 20, frame_format='frame%03d.png'):
    import os
    frame_dir = os.path.expanduser(frame_dir)
    if not os.path.isdir(frame_dir):
        raise IOError('Cannot find directory: {}'.format(frame_dir))
    if frame_dir[-1] != os.path.sep:
        frame_dir += os.path.sep
    if video_fname is None:
        parent_dir, base_dir = os.path.split(os.path.dirname(frame_dir))
        video_fname = os.path.join(parent_dir, base_dir + '.mp4')
        print 'Writing video to: {}'.format(video_fname)
    cmd = 'ffmpeg -framerate {0:d} -i {1}/{2} -q:scale 0 -c:v libx264 -vf fps=24 -pix_fmt yuv420p {3}'.format(video_fps, frame_dir, 
            frame_format, video_fname)
    import subprocess 
    subprocess.check_call([cmd],shell=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Frames to Video')
    parser.add_argument('frame_dir', help='Directory where frames are')
    parser.add_argument('-o', '--output', help='Video output name', default=None)
    parser.add_argument('-s', '--fps', help='Frames per second', default=20, type=int)
    parser.add_argument('-f', '--format', help='Frame Format String', default='frame%03d.png', type=str)
    args = parser.parse_args()
    frame_dir = args.frame_dir
    video_fname = args.output
    fps = args.fps
    frame_format = args.format
    frames_to_video(frame_dir, video_fname=video_fname, video_fps=fps, frame_format=frame_format)

