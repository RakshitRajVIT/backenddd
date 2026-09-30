const BASE=import.meta.env.VITE_API_URL || '';
export const apiUrl=path=>`${BASE}${path}`;
async function call(path, options={}){const r=await fetch(apiUrl(path),options);const body=await r.json().catch(()=>null);if(!r.ok||!body?.success)throw new Error(body?.error?.message||`Request failed (${r.status})`);return body.data;}
export async function uploadVideo(file){const form=new FormData();form.append('file',file);return call('/api/videos/upload',{method:'POST',body:form});}
export const startAnalysis=id=>call(`/api/videos/${id}/process`,{method:'POST'});
export const getStatus=id=>call(`/api/videos/${id}/status`);
export const getResults=id=>call(`/api/videos/${id}/results`);
