import * as Device from 'expo-device';

/** Total RAM in bytes, or null when the platform does not report it. */
export function totalRam(): number | null {
  return Device.totalMemory ?? null;
}

export function deviceName(): string {
  return Device.modelName ?? 'This phone';
}
