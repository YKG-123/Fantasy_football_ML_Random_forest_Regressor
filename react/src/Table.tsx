export function Table({
  json,
}: {
  json: { metadata: { last_update: string }; data: Array<object> };
}) {
  console.log(json);
  const data = json.data;
  return;
}
