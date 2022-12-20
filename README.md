# GIVE 2 POINT CLOUD
source = "pc/chess_seq1_frame_00.pcd"
target = "pc/chess_seq1_frame_150.pcd"


# GIVE THE SAVE PATH IF YOU WANT TO SAVE
save_path = "demo.png"

# END RUN THIS FUNCTION

render_rotated_frame(source, target,save_path,im_show=True)
