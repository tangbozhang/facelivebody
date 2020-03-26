package com.seeta.sdk;

import android.util.Log;

public class PointDetector
{
	static {
		Log.e("PointDetector", "Start Load");
		System.loadLibrary("PointDetector500Jni");
		Log.e("PointDetector", "Finish Load");
	}
	public long impl = 0;	// native object pointer

	private native void construct(String model);
	public native void dispose();

	public PointDetector(String model) {
		this.construct(model);
	}

	protected void finalize() throws java.lang.Throwable {
        super.finalize();
		this.dispose();
    } 

	public native void LoadModel(String model);

	public native void SetStable(boolean stable);

	public native int LandmarkNum();

	public native boolean DetectCroppedLandmarks(ImageData crop_image, Point[] landmarks);

	public native boolean DetectLandmarks(ImageData image, FaceInfo info, Point[] landmarks);

}
