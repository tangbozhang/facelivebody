package com.seeta.sdk;

public class ImageData
{
    public ImageData(int width, int height, int channels) {
        this.data = new byte[width * height * channels];
        this.width = width;
        this.height = height;
        this.channels = channels;
    }

    public ImageData(int width, int height) {
        this(width, height, 1);
    }

    public byte[] data;
    public int width;
    public int height;
    public int channels;
}
