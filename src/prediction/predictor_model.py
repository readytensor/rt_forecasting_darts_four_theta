import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from darts.models.forecasting.theta import FourTheta
from darts.utils.utils import SeasonalityMode, ModelMode, TrendMode
from darts import TimeSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the FourTheta Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "FourTheta Forecaster"

    def __init__(
        self,
        theta: int = 2,
        seasonality_period: Optional[int] = None,
        season_mode: SeasonalityMode = SeasonalityMode.ADDITIVE,
        model_mode: ModelMode = ModelMode.ADDITIVE,
        trend_mode: TrendMode = TrendMode.LINEAR,
        normalization: bool = True,
    ):
        """Construct a new FourTheta Forecaster

        Args:
            theta (int): Value of the theta parameter. Defaults to 2. Cannot be set to 0. If theta = 1, then the theta method restricts to a simple exponential smoothing (SES)
            seasonality_period (Optional[int]): User-defined seasonality period. If not set, will be tentatively inferred from the training series upon calling fit().
            season_mode (SeasonalityMode): Type of seasonality. Either SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.ADDITIVE or SeasonalityMode.NONE. Defaults to SeasonalityMode.MULTIPLICATIVE.
            model_mode (ModelMode): Type of model combining the Theta lines. Either ModelMode.ADDITIVE or ModelMode.MULTIPLICATIVE. Defaults to ModelMode.ADDITIVE.
            trend_mode (TrendMode): Type of trend to fit. Either TrendMode.LINEAR or TrendMode.EXPONENTIAL. Defaults to TrendMode.LINEAR.
            normalization (bool): If True, the data is normalized so that the mean is 1. Defaults to True.
        """
        self.theta = theta
        self.seasonality_period = seasonality_period
        self.season_mode = season_mode
        self.model_mode = model_mode
        self.trend_mode = trend_mode
        self.normalization = normalization
        self._is_trained = False
        self.models = {}
        self.data_schema = None

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by Forecaster.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """
        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency in ["secondly", "other"]:
            return "S"

    def fit(self, history: pd.DataFrame, data_schema: ForecastingSchema) -> None:
        """Fit the Forecaster to the training data.
        A separate FourTheta model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
        """
        np.random.seed(0)
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.models = {}

        for id, series in zip(all_ids, all_series):
            model = self._fit_on_series(history=series, data_schema=data_schema)
            self.models[id] = model

        self.all_ids = all_ids
        self._is_trained = True
        self.data_schema = data_schema

    def _fit_on_series(self, history: pd.DataFrame, data_schema: ForecastingSchema):
        """Fit FourTheta model to given individual series of data"""
        model = FourTheta(
            theta=self.theta,
            season_mode=self.season_mode,
            seasonality_period=self.seasonality_period,
            model_mode=self.model_mode,
            trend_mode=self.trend_mode,
            normalization=self.normalization,
        )

        series = TimeSeries.from_dataframe(history, value_cols=data_schema.target)

        model.fit(series)
        return model

    def predict(self, test_data: pd.DataFrame, prediction_col_name: str) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.data_schema.id_col)
            for id_ in self.all_ids
        ]
        # forecast one series at a time
        all_forecasts = []
        for id_, series_df in zip(self.all_ids, all_series):
            forecast = self._predict_on_series(key_and_future_df=(id_, series_df))
            forecast.insert(0, self.data_schema.id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)

        all_forecasts.rename(
            columns={self.data_schema.target: prediction_col_name}, inplace=True
        )
        return all_forecasts

    def _predict_on_series(self, key_and_future_df):
        """Make forecast on given individual series of data"""
        key, future_df = key_and_future_df

        if self.models.get(key) is not None:
            forecast = self.models[key].predict(len(future_df))
            forecast_df = forecast.pd_dataframe()
            forecast = forecast_df[self.data_schema.target]
            future_df[self.data_schema.target] = forecast.values

        else:
            # no model found - key wasnt found in history, so cant forecast for it.
            future_df = None

        return future_df

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        **hyperparameters,
    )
    model.fit(history=history, data_schema=data_schema)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)
