# MMDVS-LF

<p align="center">
    <img src="/images/recording_scheme.png" width="600">
</p>

Dynamic Vision Sensors (DVS), offer a unique advantage in control applications, due to their high temporal resolution, and asynchronous event-based data.
Still, their adoption in machine learning algorithms remains limited.
To address this gap, and promote the development of models that leverage the specific characteristics of DVS data, we introduce the Multi-Modal Dynamic-Vision-Sensor Line Following dataset (MMDVS-LF).
This comprehensive dataset, is the first to integrate multiple sensor modalities, including DVS recordings, RGB video, odometry, and Inertial Measurement Unit (IMU) data, from a small-scale standardized vehicle.
Additionally, the dataset includes eye-tracking and demographic data of drivers performing a Line Following task on a track.
With its diverse range of data, MMDVS-LF opens new opportunities for developing deep learning algorithms, and conducting data science projects across various domains, supporting innovation in autonomous systems and control applications.

## A Multi-Modal Dynamic-Vision-Sensor Line Following Dataset

## Dataset Download

We provide two flavors of datasets:

* **Noisy**: Includes frames with limited amount of noise
* **No Noise**: All frames and observations with noise are removed

> Due to constraints in storage capacity, not all datasets are available yet.
> We will upload the remaining files, as soon as possible.

The sizes in the tables refer to the uncompressed size of the downloadable archives.

<table>
    <thead>
        <tr>
            <th>Noisy</th>
            <th>128x256</th>
            <th>256x512</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>60Hz (with RGB)</td>
            <td colspan=2, align="center">136,484 Frames</td>
        </tr>
        <tr>
            <td align="center">4.92 GB</td>
            <td align="center">10.98 GB</td>
        </tr>
        <tr>
            <td rowspan=2>100Hz (DVS only)</td>
            <td colspan=2, align="center">227,375 Frames</td>
        </tr>
        <tr>
            <td align="center"><a href="https://owncloud.tuwien.ac.at/index.php/s/QWXvZwLiK7Eqom5">1.56 GB</a></td>
            <td align="center"><a href="https://owncloud.tuwien.ac.at/index.php/s/RHWC7e8MXdO1jet">4.98 GB</a></td>
        </tr>
        <tr>
            <td rowspan=2>120Hz (DVS only)</td>
            <td colspan=2, align="center">272,838</td>
        </tr>
        <tr>
            <td align="center"><a href="https://owncloud.tuwien.ac.at/index.php/s/VKKqQnRPlNIy2HW">1.73 GB</a></td>
            <td align="center"><a href="https://owncloud.tuwien.ac.at/index.php/s/qHffNsxFWPiFd1G">5.42 GB</a></td>
        </tr>
    </tbody>
</table>

Checksums:

```md5
7da517b07d6700a4cf033e5b44490243  dvs_only_256p_100hz_noisy.tar.bz2
a0f4331dbb2a9eaabc2c40514f17d4ea  dvs_only_256p_120hz_noisy.tar.bz2
7f5b91a090fd25059774825a52ebea47  dvs_only_512p_100hz_noisy.tar.bz2
3effbba0af49999c4b672ce36910a120  dvs_only_512p_120hz_noisy.tar.bz2
```

<table>
    <thead>
        <tr>
            <th>No Noise</th>
            <th>128x256</th>
            <th>256x512</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>60Hz (with RGB)</td>
            <td colspan=2, align="center">96,161 Frames</td>
        </tr>
        <tr>
            <td align="center">2.926 GB</td>
            <td align="center">7.462 GB</td>
        </tr>
        <tr>
            <td rowspan=2>100Hz (DVS only)</td>
            <td colspan=2, align="center">160,127 Frames</td>
        </tr>
        <tr>
            <td align="center"><a href="https://owncloud.tuwien.ac.at/index.php/s/nKxI4XYEdGJnD2I">1.08 GB</a></td>
            <td align="center"><a href="https://owncloud.tuwien.ac.at/index.php/s/YvQF9DJkqCAL0z5">3.45 GB</a></td>
        </tr>
        <tr>
            <td rowspan=2>120Hz (DVS only)</td>
            <td colspan=2, align="center">192,127 Frames</td>
        </tr>
        <tr>
            <td align="center"><a href="https://owncloud.tuwien.ac.at/index.php/s/R4iq4kYOnowcKBh">1.19 GB</a></td>
            <td align="center">3.75 GB</td>
        </tr>
    </tbody>
</table>

Checksums:

```md5
cc51c86dae91e8aa818ad1f42bf6092b  dvs_only_256p_100hz.tar.bz2
3bac60150f799a58da2627d54bdeb5e5  dvs_only_256p_120hz.tar.bz2
01cdc95effbf52b88c4c760a9a52f8d9  dvs_only_512p_100hz.tar.bz2
```

## Potential Use Cases of the MMDVS-LF
<p align="center">
    <img src="/images/dvs_training_options.png" width="600">
</p>

## Networks Used for Benchmarking
- CNN
- CNN + Simple RNN
- CNN + MGU
- CNN + GRU
- CNN + LSTM
- CNN + LTC

### Training Setup
<p align="center">
    <img src="/images/dvs_networks.png" width="600">
</p>
