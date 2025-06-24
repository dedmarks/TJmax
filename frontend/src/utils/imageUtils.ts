import axios from 'axios';

export const getFullImageUrl = (url: string): string => {
  if (url.startsWith('/')) {
    return `${axios.defaults.baseURL}${url}`;
  }
  return url;
};