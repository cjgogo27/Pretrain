// A logical group viewer, not a distributed runtime.
export function groupFor(groups, axis, rank) {
  if(!Number.isInteger(rank)||rank<0||!Array.isArray(groups[axis])) throw new Error('分组或rank无效');
  const found=groups[axis].filter(group=>group.includes(rank));
  if(found.length!==1) throw new Error('该rank没有唯一的分组归属');
  return found[0];
}
