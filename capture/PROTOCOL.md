# Capture protocol (one page) — follow it literally

**Before any tier:** switch on every light, open every interior door fully (flat against the wall), move people and pets out of view. Clean phone lens. Do not zoom. Walk at half your normal speed and turn slowly (a full turn takes about 10 seconds). Plan your route: start at the entrance or hallway, visit every room, and **finish where you started**.

## A. LiDAR (iPhone/iPad **Pro** only) — app: *Stray Scanner* (free, App Store)
1. Install **Stray Scanner**. Open it once and allow camera access. Leave all settings at default.
2. Stand at the start point. Tap **Record**. One recording covers the whole property — do not stop between rooms.
3. Hold the phone **upright (portrait)** at chest height, screen facing you.
4. In every room (including the hallway): walk to the middle. **Turn a full circle with the phone tilted up** so the line where walls meet the ceiling stays in view. **Then turn a full circle tilted down** so the line where walls meet the floor stays in view. If the room is longer than 3.5 m (about 4 big steps), do both circles at two spots, one in each half.
5. When you pass through a doorway, point the phone **at the door frame from the top of the frame down to the floor** for 2 seconds, on both sides.
6. Finish: go back to the start point, do one more slow full circle there, tap **Stop**.
7. Hand-off: Stray Scanner → the recording → **Share** → AirDrop/save to Files → copy the folder to the computer. Run `scan run <that folder>`.

## B. Video (any iPhone 15 or newer) — app: built-in *Camera*
1. Camera → **Video**, **1×** lens. (Settings → Camera → Record Video: 4K at 30 fps; any setting works.)
2. Hold the phone **upright (portrait)**. Tap **Record** at the start point and follow steps A4–A6 exactly (same circles, same door frames, same finish). Keep the phone moving smoothly; no sudden swings.
3. Hand-off: AirDrop the video (.MOV) to the computer. Run `scan run <the .MOV file or its folder>`.

## C. Photos (any iPhone 15 or newer) — app: built-in *Camera*
1. Camera → **Photo**, tap **0.5×** (ultra-wide). Flash off. Keep it **landscape** except for doorway shots.
2. Visit rooms in an order where **each new room is entered from a room you already photographed**. Per room take **4 to 8 photos**:
   * **Corners:** stand about half a metre from each corner, aim at the opposite corner, include floor and ceiling. (If the room has more than 4 doors, do 2 opposite corners only.)
   * **Each doorway to another room (portrait):** stand with your back to the opposite wall, aim straight at the doorway so **the whole door frame, the floor through it and the wall above it** are in the photo. Outside doors need no doorway photo.
   * **Hallways or rooms more than 2.5× longer than wide:** also photograph each end wall from 1.5 m away.
3. Hand-off: AirDrop the photos to the computer. Make **one folder per room**, named in the order you walked: `01_hall`, `02_kitchen`, … (the name after the number is optional). Put the folders in one parent folder and run `scan run <parent folder>`.

## Avoid (all tiers)
* Pointing straight at a **mirror** or glass door for long; sweep past it.
* **Dark rooms**: if a room has no light, use a lamp; the pipeline widens intervals on blurry frames but cannot invent detail.
* **Wet or glossy floors**: fine, but keep the phone tilted so walls fill most of the view.
* Covering the top camera area with fingers; walking backwards through doors.

**Time:** LiDAR or video ≈ 1 minute per room. Photos ≈ 1 minute per room. Processing: one command per capture.
