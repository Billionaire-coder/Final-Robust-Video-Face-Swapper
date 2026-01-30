import cv2
import dlib
import numpy as np
import sys
import os

# --- Configuration ---
# You MUST have this file in the same directory as the script.
# Download link is often provided on dlib's GitHub or by searching for it.
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
OUTPUT_VIDEO_NAME = "real_blended_face_swap_output.mp4"

# Initialize Dlib components
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
except RuntimeError as e:
    print("--- ERROR: DLIB MODEL MISSING ---")
    print(f"Failed to load dlib predictor: {e}")
    print("Please ensure 'shape_predictor_68_face_landmarks.dat' is in the script directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during Dlib setup: {e}")
    sys.exit(1)


# --- CORE GEOMETRY FUNCTIONS (Landmarks and Triangulation) ---

def get_face_landmarks(img, detector, predictor):
    """Detects the largest face and returns its 68 landmarks."""
    if img is None: return None, None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray, 2) 
    
    if not faces: return None, None
    
    # Select the largest face for robust tracking
    face_rect = max(faces, key=lambda rect: rect.area())
    landmarks = predictor(img_gray, face_rect)
    return face_rect, np.array([[p.x, p.y] for p in landmarks.parts()])


def calculate_delaunay_triangles(rect, points):
    """
    Calculates the Delaunay triangulation indices for a set of points 
    within a bounding rectangle.
    """
    # Create a Subdiv2D object and insert all points
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    # Get the list of triangle coordinates
    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []
    
    # Find the landmark indices corresponding to each triangle vertex
    for t in triangle_list:
        pt = [ (t[0], t[1]), (t[2], t[3]), (t[4], t[5]) ]
        indices = []
        for i in range(3):
            for j, p in enumerate(points):
                # Check for near-exact match (due to float precision)
                if abs(pt[i][0] - p[0]) < 1.0 and abs(pt[i][1] - p[1]) < 1.0:
                    indices.append(j)
                    break
        
        if len(indices) == 3:
            delaunay_triangles.append(indices)
            
    return delaunay_triangles


def warp_triangle(img_source, img_target, tri_source_pts, tri_target_pts):
    """
    Warp a triangle from img_source onto the area defined by a triangle in img_target.
    Includes robust boundary clipping for stability.
    """
    # 1. Get bounding boxes
    rect1 = cv2.boundingRect(np.float32([tri_source_pts]))
    rect2 = cv2.boundingRect(np.float32([tri_target_pts]))
    
    # Store original coordinates for relative point calculations
    (x1_orig, y1_orig, w1_orig, h1_orig) = rect1
    (x2_orig, y2_orig, w2_orig, h2_orig) = rect2
    
    img_h, img_w, _ = img_target.shape
    img_source_h, img_source_w, _ = img_source.shape 

    # --- CRITICAL FIX: Clip Source Bounding Box (rect1) to its own image boundaries ---
    x1_start = max(0, x1_orig)
    y1_start = max(0, y1_orig)
    x1_end = min(x1_orig + w1_orig, img_source_w)
    y1_end = min(y1_orig + h1_orig, img_source_h)
    
    # Calculate the clipped dimensions
    x1 = x1_start
    y1 = y1_start
    w1 = x1_end - x1_start
    h1 = y1_end - y1_start
    
    # Check if the source crop is valid (prevents the cv2.error: src.cols > 0)
    if w1 <= 1 or h1 <= 1: 
        return
    
    # 3. Target Clipping 
    x2_end = min(x2_orig + w2_orig, img_w)
    y2_end = min(y2_orig + h2_orig, img_h)
    
    x2 = max(0, x2_orig)
    y2 = max(0, y2_orig)
    
    w2 = x2_end - x2
    h2 = y2_end - y2
    
    if w2 <= 1 or h2 <= 1: 
        return

    # 4. Adjust triangle points to be relative to their ORIGINAL bounding box origin 
    tri1_cropped = []
    tri2_unclipped_relative = []
    
    for i in range(0, 3):
        tri1_cropped.append(((tri_source_pts[i][0] - x1_orig), (tri_source_pts[i][1] - y1_orig)))
        tri2_unclipped_relative.append(((tri_target_pts[i][0] - x2_orig), (tri_target_pts[i][1] - y2_orig)))


    # 5. Calculate the Affine Transformation Matrix (M)
    M = cv2.getAffineTransform(np.float32(tri1_cropped), np.float32(tri2_unclipped_relative))

    # 6. Warp the cropped source triangle (Using the clipped dimensions x1, y1, w1, h1)
    img2_warped_tri = cv2.warpAffine(
        img_source[y1:y1 + h1, x1:x1 + w1], 
        M, 
        (w2, h2), 
        None, 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REFLECT_101
    )

    # 7. Create a mask for the target triangle area 
    mask = np.zeros((h2, w2, 3), dtype=np.uint8)
    
    # Re-calculate tri2 points relative to the CLIPPED origin (x2, y2) for the mask
    tri2_cropped_for_mask = []
    for i in range(3):
        tri2_cropped_for_mask.append(((tri_target_pts[i][0] - x2), (tri_target_pts[i][1] - y2)))

    cv2.fillConvexPoly(mask, np.int32(tri2_cropped_for_mask), (255, 255, 255), 16, 0)
    
    # 8. Apply the mask to the warped triangle
    img2_warped_tri = cv2.bitwise_and(img2_warped_tri, mask)

    # 9. Place the warped and masked triangle back onto the target image
    img2_area = img_target[y2:y2 + h2, x2:x2 + w2]
    img2_area_masked = cv2.bitwise_and(img2_area, cv2.bitwise_not(mask))
    
    img_target[y2:y2 + h2, x2:x2 + w2] = cv2.add(img2_area_masked, img2_warped_tri)


def blend_final_output(warped_frame, original_frame, target_landmarks):
    """
    Applies the final seamless cloning operation for smooth edges (essential for realism).
    """
    # Create the mask for the face hull
    hull = cv2.convexHull(target_landmarks)
    hull_mask = np.zeros_like(original_frame, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, (255, 255, 255))
    
    # Find the center point of the hull 
    (x_center, y_center, w, h) = cv2.boundingRect(hull)
    center_point = (int(x_center + w // 2), int(y_center + h // 2))

    return cv2.seamlessClone(
        warped_frame, 
        original_frame, 
        hull_mask, 
        center_point, 
        cv2.NORMAL_CLONE # NORMAL_CLONE is the best for blending skin
    )

def color_correct_source(img_source, source_landmarks, img_target, target_landmarks):
    """
    Adjusts the color (mean and standard deviation) of the source face 
    to match the target face for seamless blending and realistic skin tone.
    """
    # 1. Create a mask for the face area in both images
    hull_source = cv2.convexHull(source_landmarks)
    mask_source = np.zeros(img_source.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_source, hull_source, 255)

    hull_target = cv2.convexHull(target_landmarks)
    mask_target = np.zeros(img_target.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_target, hull_target, 255)

    # 2. Get the masked BGR pixel data
    masked_source_pixels = img_source[mask_source == 255]
    masked_target_pixels = img_target[mask_target == 255]

    if masked_source_pixels.size == 0 or masked_target_pixels.size == 0:
        print("Warning: Insufficient pixels for color correction. Skipping.")
        return img_source

    # 3. Calculate mean and standard deviation for BGR channels
    mean_source = np.mean(masked_source_pixels, axis=0)
    std_source = np.std(masked_source_pixels, axis=0)
    
    mean_target = np.mean(masked_target_pixels, axis=0)
    std_target = np.std(masked_target_pixels, axis=0)
    
    # Avoid division by zero
    std_source[std_source < 1.0] = 1.0

    # 4. Apply transformation (Color Transfer)
    img_source_float = img_source.astype(np.float32)
    
    # adjusted_pixel = ((source_pixel - source_mean) * (target_std / source_std)) + target_mean
    img_source_float = (img_source_float - mean_source)
    img_source_float = img_source_float * (std_target / std_source)
    img_source_float = img_source_float + mean_target
    
    img_source_corrected = np.clip(img_source_float, 0, 255).astype(np.uint8)

    return img_source_corrected


# --- MAIN VIDEO PROCESSING LOOP ---

def process_video_swap(video_path, source_img_path):
    """
    Main function to initialize, process, and write the swapped video.
    """
    # 1. Load Source Image and Landmarks ONCE
    img_source = cv2.imread(source_img_path)
    if img_source is None:
        print(f"Error: Could not load source image at {source_img_path}")
        return

    print("Analyzing source face...")
    source_rect, source_landmarks = get_face_landmarks(img_source, detector, predictor)
    if source_landmarks is None:
        print("Error: No face detected in the source image. Please choose a clear source image.")
        return

    # 2. Open Target Video and Get Properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 3. Search for a face to Initialize Geometry and Color
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    search_limit = min(frame_count_total, 100) 
    initial_target_landmarks = None
    initial_target_frame = None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in range(search_limit):
        ret, search_frame = cap.read()
        if not ret: break
        
        _, landmarks = get_face_landmarks(search_frame, detector, predictor)
        
        if landmarks is not None:
            initial_target_landmarks = landmarks
            initial_target_frame = search_frame.copy()
            print(f"Face found in Frame {i + 1}. Initializing geometry and color model.")
            break
            
    if initial_target_landmarks is None:
        print(f"Error: No face was detected within the first {search_limit} frames of the video. Cannot proceed.")
        cap.release()
        return

    # 4. CRITICAL STEP: Color Correction for Pro Blending
    print("Applying color and contrast correction to source image for blending...")
    img_source = color_correct_source(
        img_source, source_landmarks, 
        initial_target_frame, initial_target_landmarks
    )
    
    # 5. Calculate fixed Delaunay Triangles 
    (x, y, w, h) = cv2.boundingRect(initial_target_landmarks)
    target_bbox = (x, y, x + w, y + h) 
    delaunay_triangles = calculate_delaunay_triangles(target_bbox, initial_target_landmarks)
    print(f"Calculated {len(delaunay_triangles)} consistent triangles.")


    # 6. Initialize Video Writer 
    # 'mp4v' or 'XVID' in an MP4 container is the most universally compatible choice for this library.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, fps, (frame_width, frame_height))
    print(f"Starting frame-by-frame video processing at {fps:.2f} FPS. Output: {OUTPUT_VIDEO_NAME}")

    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # 7. Locate the current target face landmarks
        _, current_target_landmarks = get_face_landmarks(frame, detector, predictor)

        if current_target_landmarks is None:
            # If face is lost, write the original frame to maintain video flow
            out.write(frame)
            continue
        
        frame_warped = frame.copy()
        
        # 8. Warp Each Triangle 
        for tri_indices in delaunay_triangles:
            try:
                tri_source_pts = np.array([source_landmarks[i] for i in tri_indices], dtype=np.int32)
                tri_target_pts = np.array([current_target_landmarks[i] for i in tri_indices], dtype=np.int32)
                
                warp_triangle(img_source, frame_warped, tri_source_pts, tri_target_pts)
            except Exception as e:
                # Catch any residual error during warping and continue to the next frame
                print(f"Skipping triangle in frame {frame_count} due to error: {e}")
                continue

        # 9. Final Blending (Seamless Clone)
        final_frame = blend_final_output(frame_warped, frame, current_target_landmarks)
        
        # 10. Write the swapped frame
        out.write(final_frame)

    # 11. Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n--- PROCESSING COMPLETE ---")
    print(f"Successfully processed {frame_count} frames.")
    print(f"Video saved as: {OUTPUT_VIDEO_NAME}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Welcome to the ROBUST VIDEO FACE SWAPPER.")
    print("This script is optimized for blending and stability.")
    
    # Get user input for files
    video_path = input("Enter the full path to the TARGET VIDEO file: ")
    source_img_path = input("Enter the full path to the SOURCE FACE image file: ")

    # Run the video processing function
    process_video_swap(video_path, source_img_path)