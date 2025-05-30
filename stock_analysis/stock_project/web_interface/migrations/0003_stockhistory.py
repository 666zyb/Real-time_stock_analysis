# Generated by Django 5.1.7 on 2025-03-20 11:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web_interface', '0002_remove_stockrealtimedata_high_price'),
    ]

    operations = [
        migrations.CreateModel(
            name='StockHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stock_code', models.CharField(max_length=20)),
                ('date', models.DateField()),
                ('open_price', models.FloatField()),
                ('close_price', models.FloatField()),
                ('high_price', models.FloatField()),
                ('low_price', models.FloatField()),
                ('volume', models.IntegerField()),
            ],
            options={
                'db_table': '江西铜业_history',
            },
        ),
    ]
