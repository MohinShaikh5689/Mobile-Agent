// eslint-disable-next-line @typescript-eslint/no-require-imports
const { applySigning } = require('../plugins/withReleaseSigning');

// The signing part of app/build.gradle as `expo prebuild` generates it (SDK 58 template).
const TEMPLATE = `    signingConfigs {
        debug {
            storeFile file('debug.keystore')
            storePassword 'android'
            keyAlias 'androiddebugkey'
            keyPassword 'android'
        }
    }
    buildTypes {
        debug {
            signingConfig signingConfigs.debug
        }
        release {
            // Caution! In production, you need to generate your own keystore file.
            // see https://reactnative.dev/docs/signed-apk-android.
            signingConfig signingConfigs.debug
            minifyEnabled enableMinifyInReleaseBuilds
        }
    }`;

describe('withReleaseSigning', () => {
  it('adds an env-driven release config and uses it only for release builds', () => {
    const out: string = applySigning(TEMPLATE);
    expect(out).toContain("def ksFile = System.getenv('ANDROID_KEYSTORE_FILE')");
    expect(out).toContain(
      "signingConfig System.getenv('ANDROID_KEYSTORE_FILE') ? signingConfigs.release : signingConfigs.debug",
    );
    // The debug build type still uses the debug key.
    expect(out).toMatch(/debug \{\s*\n\s*signingConfig signingConfigs\.debug/);
  });

  it('is idempotent', () => {
    const once: string = applySigning(TEMPLATE);
    expect(applySigning(once)).toBe(once);
  });

  it('fails loudly when the template changes', () => {
    expect(() => applySigning('android { }')).toThrow(/no longer matches/);
  });
});
