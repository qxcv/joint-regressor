# Pretrained VGGNet16 for Keras

This was downloaded from [this
gist](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) by
`baraldilorenzo` on Github. It's not clear that he's doing dropout correctly
(Keras previously applied dropout scaling at test rather than exclusively at
training time, but now applies it only at training time like Caffe has always
done).

You can now get the weights from the URL in `download_urls.txt` (in this
directory). I will try to keep that URL up so that this code doesn't rot (even
if the aforementioned gist gets taken down).
