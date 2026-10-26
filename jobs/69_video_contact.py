"""Contact sheet of the explainer video (one frame every ~6 s) for a visual check."""
import imageio.v2 as imageio
from PIL import Image
r = imageio.get_reader("/export/home/kaziz/g05/results/video/g05_explainer.mp4")
n = r.count_frames(); idx = list(range(60, n, 120))[:16]
ims = [Image.fromarray(r.get_data(i)).resize((640, 360)) for i in idx]
sheet = Image.new("RGB", (640 * 4, 360 * ((len(ims) + 3) // 4)), (255, 255, 255))
for k, im in enumerate(ims):
    sheet.paste(im, ((k % 4) * 640, (k // 4) * 360))
sheet.save("/export/home/kaziz/g05/results/video/contact.jpg", quality=85); print(n, idx)
