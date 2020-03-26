package com.seeta.sdk;

import android.util.Log;

public class FaceDetector
{
	static {
		System.loadLibrary("FaceDetector512Jni");
	}

	public long impl = 0;	// native object pointer

	private native void construct(String model);
	private native void construct(String model, String device);
	public native void dispose();

	public FaceDetector(String model) {
		this.construct(model);
	}

	public FaceDetector(String model, String device) {
		this.construct(model, device);
	}

	protected void finalize() throws java.lang.Throwable {
        super.finalize();
		this.dispose();
    }  

	public native FaceInfo[] Detect(ImageData img);
	
	public native void SetMinFaceSize(int size);
	public native void SetImagePyramidScaleFactor(float factor);
	public native void SetScoreThresh(float thresh1, float thresh2, float thresh3);
	public void SetVideoStable() {
		SetVideoStable(true);
	}
	public native void SetVideoStable(boolean stable);
	public native boolean GetVideoStable();


	public native void SetNumThreads(int num);
}
