package com.seeta.sdk;

import android.util.Log;

public class FaceRecognizer
{
    static {
        System.loadLibrary("FaceRecognizer514Jni");
    }

	public long impl = 0;	// native object pointer

	private native void construct(String model);
	private native void construct(String model, String device);
	public native void dispose();

    public FaceRecognizer() {
        this("");
    }

	public FaceRecognizer(String model) {
		this.construct(model);
	}

	public FaceRecognizer(String model, String device) {
		this.construct(model, device);
	}

	protected void finalize() throws java.lang.Throwable {
        super.finalize();
		this.dispose();
    }

    public native boolean LoadModel(String model);

    public native boolean LoadModel(String model, String device);

    public native int GetFeatureSize();

    public native int GetCropWidth();

    public native int GetCropHeight();

    public native int GetCropChannels();

    public boolean CropFace(ImageData image, Point[] landmarks, ImageData face) {
        return CropFace(image, landmarks, face, 1);
    }

    public native boolean CropFace(ImageData image, Point[] landmarks, ImageData face, int pos_num);

    public native boolean ExtractFeature(ImageData face, float[] feats);

    public native boolean ExtractFeatureNormalized(ImageData face, float[] feats);

    public boolean ExtractFeatureWithCrop(ImageData image, Point[] landmarks, float[] feats) {
        return ExtractFeatureWithCrop(image, landmarks, feats, 1);
    }

    public native boolean ExtractFeatureWithCrop(ImageData image, Point[] landmarks, float[] feats, int pos_num);

    public boolean ExtractFeatureWithCropNormalized(ImageData image, Point[] landmarks, float[] feats) {
        return ExtractFeatureWithCropNormalized(image, landmarks, feats, 1);
    }

    public native boolean ExtractFeatureWithCropNormalized(ImageData image, Point[] landmarks, float[] feats, int pos_num);

    public float CalcSimilarity(float[] fc1, float[] fc2) {
        return CalcSimilarity(fc1, fc2, -1);
    }

    public native float CalcSimilarity(float[] fc1, float[] fc2, long dim);

    public float CalcSimilarityNormalized(float[] fc1, float[] fc2) {
        return CalcSimilarityNormalized(fc1, fc2, -1);
    }

    public native float CalcSimilarityNormalized(float[] fc1, float[] fc2, long dim);

    public static native void SetNumThreads(int num);

    public static native int SetMaxBatchGlobal(int max_batch);

    public native int GetMaxBatch();

    public static native int SetCoreNumberGlobal(int core_number);

    public native int GetCoreNumber();

    public native boolean ExtractFeature(ImageData[] faces, float[] feats);

    public native boolean ExtractFeatureNormalized(ImageData[] faces, float[] feats);

    public native boolean ExtractFeatureWithCrop(ImageData[] images, Point[] points, float[] feats);

    public native boolean ExtractFeatureWithCropNormalized(ImageData[] images, Point[] points, float[] feats);
}
