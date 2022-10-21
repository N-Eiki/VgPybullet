# Making a textured model for simulation

## Making a 3D model
1. Install [AUTODESK Fusion360](https://www.autodesk.co.jp/products/fusion-360/overview)
2. Make a stl file like below figure

## Making textured model
1. Install [Blendar](https://blender.jp/), [MeshLab](https://www.meshlab.net/), and [GIMP](https://www.gimp.org/)

2. Open Blendar

3. Revise Author as follows: File->User Preference->File tab->Author->Save User Settings

4. Import stl file as follows: File->Import

5. Set Unit as Metric and scale=1.0

6. Set Material by clicking New and set Intensit as 1.0 and Emit as 0.45

7. Mark seam as follows: Shift+Right click for edges in Edit mode->Ctrl+E->Mark seam

8. UV unwrapping as follows: click A in Edit mode->Mesh->UV unwrap->Unwrap

9. Export UV layout as follows: click UVs->Export UV layout->XXX.png

10. Make a texture file with GIMP

11. Apply texture in Blendar as follows: Open in UV Editing and Click Texture tab->New->Image or Movie->Open texture image

12. Choose UV for Coordinate and UVMap for Map in Mapping tab

13. Check Rendered view
