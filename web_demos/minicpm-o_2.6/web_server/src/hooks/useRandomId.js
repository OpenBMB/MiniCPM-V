const uid = 'uid';
export const setNewUserId = () => {
    const randomId = Math.random().toString(36).slice(2).toUpperCase();
    localStorage.setItem(uid, randomId);
    return randomId;
};
export const getNewUserId = () => {
    return localStorage.getItem('uid');
};
