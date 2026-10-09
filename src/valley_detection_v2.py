# # this is valley code original 09-07-26
#
# import numpy as np
# from matplotlib import pyplot as plt
# from skimage.morphology import skeletonize
#
# # image_name = "h4"
#
# # ==========================================================
# # Input Paths
# # ==========================================================
#
# # raw_path = f"data/raw/hirise_original/{image_name}.jpg"
# # pred_path = f"outputs/prediction/{image_name}_predicted.png"
#
# import os
# import cv2
#
# # ==========================================================
# # Input Image
# # ==========================================================
#
# image_name = "h3"      # Enter filename WITHOUT extension
#
# input_folder = "data/raw/hirise_original"
#
# supported_formats = [
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".tif",
#     ".tiff",
#     ".bmp"
# ]
#
# raw_path = None
#
# # Search for the image in any supported format
# for ext in supported_formats:
#
#     candidate = os.path.join(
#         input_folder,
#         image_name + ext
#     )
#
#     if os.path.exists(candidate):
#         raw_path = candidate
#         break
#
# if raw_path is None:
#     raise FileNotFoundError(
#         f"No image found for '{image_name}' "
#         f"in {input_folder}"
#     )
#
# # ==========================================================
# # Convert to JPG (temporary)
# # ==========================================================
#
# temp_jpg = os.path.join(
#     input_folder,
#     f"{image_name}_temp.jpg"
# )
#
# img = cv2.imread(raw_path)
#
# cv2.imwrite(temp_jpg, img)
#
# # Use the temporary JPG for processing
# raw_path = temp_jpg
#
# # Predicted mask path
# pred_path = os.path.join(
#     "outputs",
#     "prediction",
#     f"{image_name}_predicted.png"
# )
#
# # ==========================================================
# # Output Folder
# # ==========================================================
#
# output_folder = "outputs/valley_detection"
# os.makedirs(output_folder, exist_ok=True)
#
# binary_path = os.path.join(output_folder, f"{image_name}_binary.png")
# clean_path = os.path.join(output_folder, f"{image_name}_clean.png")
# valley_path = os.path.join(output_folder, f"{image_name}_valleys.png")
# skeleton_path = os.path.join(output_folder, f"{image_name}_skeleton.png")
# result_path = os.path.join(output_folder, f"{image_name}_results.png")
#
#
# # ==========================================================
# # Load Images
# # ==========================================================
#
# raw = cv2.imread(raw_path)
# pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
#
# if raw is None:
#     raise FileNotFoundError(raw_path)
#
# if pred is None:
#     raise FileNotFoundError(pred_path)
#
# # ==========================================================
# # Threshold Prediction
# # ==========================================================
#
# _, binary = cv2.threshold(
#     pred,
#     127,
#     255,
#     cv2.THRESH_BINARY
# )
#
# cv2.imwrite(binary_path, binary)
#
# # ==========================================================
# # Morphological Cleaning
# # ==========================================================
#
# kernel = cv2.getStructuringElement(
#     cv2.MORPH_ELLIPSE,
#     (5,5)
# )
#
# clean = cv2.morphologyEx(
#     binary,
#     cv2.MORPH_OPEN,
#     kernel
# )
#
# clean = cv2.morphologyEx(
#     clean,
#     cv2.MORPH_CLOSE,
#     kernel
# )
#
# cv2.imwrite(clean_path, clean)
#
# # ==========================================================
# # Find Contours
# # ==========================================================
#
# contours, _ = cv2.findContours(
#     clean,
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )
#
# # ==========================================================
# # Smart Valley Filtering
# # ==========================================================
#
# MIN_AREA = 150
# MIN_ASPECT_RATIO = 2.5
# MAX_CIRCULARITY = 0.45
#
# filtered = []
#
# print("\n==============================================")
# print("        Candidate Valley Analysis")
# print("==============================================\n")
#
# for cnt in contours:
#
#     area = cv2.contourArea(cnt)
#
#     # Ignore very small objects
#     if area < MIN_AREA:
#         continue
#
#     perimeter = cv2.arcLength(cnt, True)
#
#     if perimeter == 0:
#         continue
#
#     x, y, w, h = cv2.boundingRect(cnt)
#
#     margin = 5  # pixels
#
#     # Ignore contours touching image borders
#     if (
#             x <= margin or
#             y <= margin or
#             x + w >= clean.shape[1] - margin or
#             y + h >= clean.shape[0] - margin
#     ):
#         continue
#     # Orientation using fitted ellipse
#     orientation = 0
#
#     if len(cnt) >= 5:
#         ellipse = cv2.fitEllipse(cnt)
#
#         orientation = ellipse[2]
#
#     # valley["orientation"] = orientation
#
#     # Long and narrow structures have higher aspect ratio
#     aspect_ratio = max(w, h) / (min(w, h) + 1)
#
#     # Circularity
#     circularity = (4 * np.pi * area) / (perimeter * perimeter)
#
#     print(
#         f"Area={area:7.1f} | "
#         f"Aspect={aspect_ratio:5.2f} | "
#         f"Circularity={circularity:5.2f} | "
#         f"Bounding Box={w}x{h}"
#     )
#
#
#
#     if (
#             area >= MIN_AREA and
#             aspect_ratio >= MIN_ASPECT_RATIO and
#             circularity <= MAX_CIRCULARITY
#     ):
#
#         # ------------------------------------------------------
#         # Confidence Score
#         # ------------------------------------------------------
#
#         confidence = 0
#
#         if aspect_ratio >= 5:
#             confidence += 40
#
#         elif aspect_ratio >= 3:
#             confidence += 25
#
#         if circularity <= 0.20:
#             confidence += 40
#
#         elif circularity <= 0.35:
#             confidence += 25
#
#         if area >= 300:
#             confidence += 20
#
#         confidence = min(confidence, 100)
#
#         filtered.append({
#
#             "contour": cnt,
#
#             "area": area,
#
#             "perimeter": perimeter,
#
#             "x": x,
#             "y": y,
#             "w": w,
#             "h": h,
#
#             "aspect_ratio": aspect_ratio,
#
#             "circularity": circularity,
#
#             "confidence": confidence
#
#         })
#
# print("\n----------------------------------------------")
# print(f"Valley Candidates Selected : {len(filtered)}")
# print("----------------------------------------------")
#
# # ==========================================================
# # Draw Detected Valleys
# # ==========================================================
#
# overlay = raw.copy()
#
# for i, valley in enumerate(filtered):
#
#     cnt = valley["contour"]
#
#     x = valley["x"]
#     y = valley["y"]
#     w = valley["w"]
#     h = valley["h"]
#
#
#
#     # Dynamic contour thickness
#     if w > 250 or h > 250:
#         thickness = 3
#     elif w > 120 or h > 120:
#         thickness = 2
#     else:
#         thickness = 1
#
#
#
#     # Draw bounding box
#     cv2.rectangle(
#         overlay,
#         (x, y),
#         (x + w, y + h),
#         (255, 0, 0),
#         2
#     )
#
#
#
# cv2.imwrite(valley_path, overlay)
#
# # ==========================================================
# # Skeleton Generation
# # ==========================================================
#
# binary_bool = clean > 0
#
# skeleton = skeletonize(binary_bool)
#
# skeleton = (skeleton * 255).astype(np.uint8)
#
# cv2.imwrite(skeleton_path, skeleton)
#
#
# # ==========================================================
# # Valley Length Estimation
# # ==========================================================
#
# PIXEL_SIZE = 0.25      # meters per pixel
#
# for valley in filtered:
#
#     valley_mask = np.zeros_like(clean)
#
#     cv2.drawContours(
#         valley_mask,
#         [valley["contour"]],
#         -1,
#         255,
#         thickness=cv2.FILLED
#     )
#
#     valley_skeleton = cv2.bitwise_and(
#         skeleton,
#         valley_mask
#     )
#
#     length_pixels = cv2.countNonZero(valley_skeleton)
#
#     valley["length"] = length_pixels * PIXEL_SIZE
#
# # ==========================================================
# # Valley Statistics
# # ==========================================================
#
# print("\n==============================================")
# print("           Valley Detection Results")
# print("==============================================\n")
#
# print(f"Detected Valleys : {len(filtered)}\n")
#
# for i, valley in enumerate(filtered):
#
#     print(f"Valley {i+1}")
#     print("--------------------------------------")
#     # print(f"Area         : {valley['area']:.2f}")
#     # print(f"Perimeter    : {valley['perimeter']:.2f}")
#     # ----------------------------------------------------------
#     # Conversion (Assuming HiRISE resolution = 0.25 m/pixel)
#     # ----------------------------------------------------------
#
#     PIXEL_SIZE = 0.25  # meters per pixel
#
#     area_m2 = valley["area"] * (PIXEL_SIZE ** 2)
#     perimeter_m = valley["perimeter"] * PIXEL_SIZE
#
#     print(f"Area         : {area_m2:.2f} m²")
#     print(f"Perimeter    : {perimeter_m:.2f} m")
#     print(
#         f"Bounding Box : "
#         f"({valley['x']},{valley['y']}) "
#         f"{valley['w']}x{valley['h']}"
#     )
#     print(f"Aspect Ratio : {valley['aspect_ratio']:.2f}")
#     print(f"Circularity  : {valley['circularity']:.2f}")
#     print(f"Confidence   : {valley['confidence']}%")
#     print()
#
# # ==========================================================
# # Visualization
# # ==========================================================
#
# plt.figure(figsize=(16, 10))
#
# # Original Image
# plt.subplot(2, 3, 1)
# plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
# plt.axis("off")
#
# # Predicted Mask
# plt.subplot(2, 3, 2)
# plt.imshow(pred, cmap="gray")
# plt.title("Predicted Mask")
# plt.axis("off")
#
# # Binary Mask
# plt.subplot(2, 3, 3)
# plt.imshow(binary, cmap="gray")
# plt.title("Binary Mask")
# plt.axis("off")
#
# # Clean Mask
# plt.subplot(2, 3, 4)
# plt.imshow(clean, cmap="gray")
# plt.title("Clean Mask")
# plt.axis("off")
#
# # Valley Detection
# plt.subplot(2, 3, 5)
#
# plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
#
# for i, valley in enumerate(filtered):
#
#     cnt = valley["contour"].squeeze()
#
#     plt.plot(
#         cnt[:, 0],
#         cnt[:, 1],
#         color="lime",
#         linewidth=1
#     )
#
#     plt.text(
#         valley["x"],
#         valley["y"] - 5,
#         f"V{i+1}",
#         color="yellow",
#         fontsize=8,
#         weight="bold"
#     )
#
# plt.title(f"Detected Valleys ({len(filtered)})")
# plt.axis("off")
#
# # Skeleton
# plt.subplot(2, 3, 6)
# plt.imshow(skeleton, cmap="gray")
# plt.title("Skeleton")
# plt.axis("off")
#
# plt.tight_layout()
#
# # ==========================================================
# # Save Figure
# # ==========================================================
#
# # plt.savefig(
# #     result_path,
# #     dpi=300,
# #     bbox_inches="tight"
# # )
# plt.savefig(
#     result_path,
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()
# plt.close()
# # plt.show()
# # plt.close()
#
# # ==========================================================
# # Final Summary
# # ==========================================================
# #
# # print("\n==============================================")
# # print(" Valley Detection Completed Successfully")
# # print("==============================================")
#
# print(f"\nImage Name          : {image_name}")
# print(f"Total Contours      : {len(contours)}")
# print(f"Detected Valleys    : {len(filtered)}")
#
# print("\nSaved Files")
#
# print("----------------------------------------------")
# print(binary_path)
# print(clean_path)
# print(valley_path)
# print(skeleton_path)
# print(result_path)
# print("----------------------------------------------")
#
# # print("\nProgram Finished Successfully.\n")
# # plt.show()
# #  plt.close(figure)
#
# print("\n==============================================")
# print(" Valley Detection Completed Successfully")
# print("==============================================")
#
# # ==========================================================
# # Remove Temporary JPG
# # ==========================================================
#
# if os.path.exists(temp_jpg):
#     os.remove(temp_jpg)
#
