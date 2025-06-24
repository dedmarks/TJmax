import axios from 'axios';

const API_URL = 'http://localhost:5000/api/upload';

export interface UploadedImage {
  url: string;
  filename: string;
  originalName: string;
}

// Upload image from a File object
export const uploadImage = async (file: File): Promise<UploadedImage> => {
  const formData = new FormData();
  formData.append('image', file);
  
  const response = await axios.post(`${API_URL}/image`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  
  return response.data;
};

// Upload image from a base64 string (for clipboard pasting)
export const uploadImageFromBase64 = async (base64Data: string, filename = 'pasted-image.png'): Promise<UploadedImage> => {
  // Convert base64 to blob
  const byteString = atob(base64Data.split(',')[1]);
  const mimeString = base64Data.split(',')[0].split(':')[1].split(';')[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  
  const blob = new Blob([ab], { type: mimeString });
  const file = new File([blob], filename, { type: mimeString });
  
  return uploadImage(file);
};