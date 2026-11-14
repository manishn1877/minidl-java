package minidl;

import java.io.IOException;

/**
 * Looks at the training data with your own eyes: prints one picture for every class as text, so you can SEE that
 * the label numbers match the letters (the class called "B" must look like a B, upright, not sideways).
 *
 *   java -cp out minidl.DataCheck &lt;emnist-folder&gt;                 letters only (26 classes, A-Z)
 *   java -cp out minidl.DataCheck &lt;emnist-folder&gt; &lt;mnist-folder&gt;   digits + letters as the combined model sees them (36 classes)
 *   java -cp out minidl.DataCheck &lt;emnist-folder&gt; --byclass            the 62 classes with capital and small letters (0-9, A-Z, a-z)
 *
 * If a picture does not look like the character printed above it, the label mapping or the picture direction is wrong:
 * do not train until it is fixed. (EMNIST letters: labels 1..26 = A..Z, pictures stored sideways; EmnistLoader corrects both.)
 */
public final class DataCheck {
    private DataCheck() {}

    public static void main(String[] args) throws IOException {
        if (args.length < 1) { System.out.println("usage: java -cp out minidl.DataCheck <emnist-folder> [mnist-folder]"); return; }
        ByteSource src;
        String chars;
        if (args.length > 1 && args[1].equals("--byclass")) { src = EmnistLoader.byClass(args[0], "test", 30000, false); chars = EmnistLoader.FULL; }
        else if (args.length > 1) { src = Alnum.load(args[1], args[0], "test", 6000); chars = Alnum.CHARS; }
        else { src = EmnistLoader.lettersFrom(args[0], "test", Integer.MAX_VALUE); chars = EmnistLoader.LETTERS; }
        byte[] px = src.copyPixels(), lab = src.copyLabels();
        int classes = chars.length();
        int[] first = new int[classes], count = new int[classes];
        java.util.Arrays.fill(first, -1);
        for (int i = 0; i < lab.length; i++) { int c = lab[i]; if (first[c] < 0) first[c] = i; count[c]++; }
        System.out.println(lab.length + " test pictures, " + classes + " classes. One picture per class (label number, character):\n");
        for (int row = 0; row < classes; row += 4) {
            StringBuilder head = new StringBuilder();
            for (int c = row; c < Math.min(classes, row + 4); c++) head.append(String.format("%-30s", "class " + c + " = '" + chars.charAt(c) + "'  (" + count[c] + " pictures)").substring(0, 30));
            System.out.println(head);
            for (int y = 0; y < 28; y += 2) {
                StringBuilder line = new StringBuilder();
                for (int c = row; c < Math.min(classes, row + 4); c++) {
                    for (int x = 0; x < 28; x++) {
                        if (first[c] < 0) { line.append(' '); continue; }
                        int v = ((px[first[c] * 784 + y * 28 + x] & 0xFF) + (px[first[c] * 784 + (y + 1) * 28 + x] & 0xFF)) / 2;
                        line.append(v > 170 ? '#' : v > 100 ? '+' : v > 40 ? '.' : ' ');
                    }
                    line.append("  ");
                }
                System.out.println(line);
            }
            System.out.println();
        }
        for (int c = 0; c < classes; c++) if (count[c] == 0) System.out.println("WARNING: class " + c + " ('" + chars.charAt(c) + "') has no pictures");
    }
}
