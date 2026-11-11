/**
 * Signs Android release builds with your own keystore when one is provided through environment
 * variables, instead of the React Native template's public debug key.
 *
 * Google's Android OAuth client is bound to the package name + the SHA-1 of the signing key, so the
 * APK must always be signed with the same, private key.
 *
 *   ANDROID_KEYSTORE_FILE      absolute path to the .jks/.keystore
 *   ANDROID_KEYSTORE_PASSWORD  store password
 *   ANDROID_KEY_ALIAS          key alias
 *   ANDROID_KEY_PASSWORD       key password
 *
 * Without ANDROID_KEYSTORE_FILE, release builds keep the debug key (fine for local testing only).
 */
const { withAppBuildGradle } = require('expo/config-plugins');

const MARKER = '// mobile-agent: release signing';

const SIGNING_CONFIG = `signingConfigs {
        ${MARKER}
        release {
            def ksFile = System.getenv('ANDROID_KEYSTORE_FILE')
            if (ksFile) {
                storeFile file(ksFile)
                storePassword System.getenv('ANDROID_KEYSTORE_PASSWORD')
                keyAlias System.getenv('ANDROID_KEY_ALIAS')
                keyPassword System.getenv('ANDROID_KEY_PASSWORD')
            }
        }`;

const RELEASE_LINE_FROM = /(release \{\s*\n(?:\s*\/\/.*\n)*\s*)signingConfig signingConfigs\.debug/;
const RELEASE_LINE_TO =
  "$1signingConfig System.getenv('ANDROID_KEYSTORE_FILE') ? signingConfigs.release : signingConfigs.debug";

function applySigning(gradle) {
  if (gradle.includes(MARKER)) return gradle;
  if (!gradle.includes('signingConfigs {') || !RELEASE_LINE_FROM.test(gradle)) {
    throw new Error(
      'withReleaseSigning: app/build.gradle no longer matches the expected template; update plugins/withReleaseSigning.js.',
    );
  }
  return gradle.replace('signingConfigs {', SIGNING_CONFIG).replace(RELEASE_LINE_FROM, RELEASE_LINE_TO);
}

module.exports = function withReleaseSigning(config) {
  return withAppBuildGradle(config, (c) => {
    if (c.modResults.language !== 'groovy') {
      throw new Error('withReleaseSigning: expected a Groovy app/build.gradle.');
    }
    c.modResults.contents = applySigning(c.modResults.contents);
    return c;
  });
};

module.exports.applySigning = applySigning;
